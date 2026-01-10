//! Trust store installation for various operating systems.
//!
//! Attempts to automatically install the CA certificate into the system
//! trust store, similar to how Caddy and mkcert work.

use std::path::Path;
use std::process::Command;

use tracing::{debug, info, warn};

/// Result of attempting to install CA into trust store.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrustResult {
    /// CA was successfully installed to the trust store
    Installed,
    /// CA was already installed
    AlreadyInstalled,
    /// Manual installation required (with reason)
    ManualRequired(String),
    /// Installation failed (with error message)
    Failed(String),
}

impl TrustResult {
    /// Returns true if the CA is trusted (either just installed or already was).
    pub fn is_trusted(&self) -> bool {
        matches!(self, TrustResult::Installed | TrustResult::AlreadyInstalled)
    }
}

/// Handles trust store operations for different platforms.
pub struct TrustStore;

impl TrustStore {
    /// Attempt to install a CA certificate into the system trust store.
    ///
    /// This tries to use platform-specific methods to install the certificate.
    /// On success, browsers and other applications will trust certificates
    /// signed by this CA.
    pub fn install_ca(ca_path: &Path) -> TrustResult {
        if !ca_path.exists() {
            return TrustResult::Failed(format!("CA certificate not found: {:?}", ca_path));
        }

        #[cfg(target_os = "macos")]
        return Self::install_macos(ca_path);

        #[cfg(target_os = "windows")]
        return Self::install_windows(ca_path);

        #[cfg(target_os = "linux")]
        return Self::install_linux(ca_path);

        #[cfg(not(any(target_os = "macos", target_os = "windows", target_os = "linux")))]
        TrustResult::ManualRequired("Unsupported operating system".to_string())
    }

    /// Install CA on macOS using the security command.
    #[cfg(target_os = "macos")]
    fn install_macos(ca_path: &Path) -> TrustResult {
        // Get the user's login keychain path
        let keychain = match Self::get_macos_login_keychain() {
            Ok(k) => k,
            Err(e) => return TrustResult::Failed(e),
        };

        info!("Installing CA to macOS keychain: {}", keychain);

        // Check if already trusted by verifying trust settings (not just certificate existence)
        let check_output = Command::new("security")
            .args(["dump-trust-settings", "-d"])
            .output();

        if let Ok(output) = check_output {
            let stdout = String::from_utf8_lossy(&output.stdout);
            if stdout.contains("Alloy Local Development CA") {
                debug!("CA certificate already trusted in keychain");
                return TrustResult::AlreadyInstalled;
            }
        }

        // Install the certificate with trust settings
        // Use -p ssl to specifically trust for SSL/TLS
        let output = Command::new("security")
            .args([
                "add-trusted-cert",
                "-r",
                "trustRoot",
                "-p",
                "ssl",
                "-k",
                &keychain,
            ])
            .arg(ca_path)
            .output();

        match output {
            Ok(output) => {
                if output.status.success() {
                    info!("CA certificate installed to macOS keychain");
                    TrustResult::Installed
                } else {
                    let stderr = String::from_utf8_lossy(&output.stderr);
                    if stderr.contains("already exists") || stderr.contains("duplicate") {
                        TrustResult::AlreadyInstalled
                    } else {
                        warn!("Failed to install CA: {}", stderr);
                        TrustResult::ManualRequired(format!(
                            "Could not auto-install CA: {}",
                            stderr.trim()
                        ))
                    }
                }
            }
            Err(e) => TrustResult::Failed(format!("Failed to run security command: {}", e)),
        }
    }

    /// Get the path to the user's login keychain on macOS.
    #[cfg(target_os = "macos")]
    fn get_macos_login_keychain() -> Result<String, String> {
        // Try to get the default keychain
        let output = Command::new("security")
            .args(["default-keychain"])
            .output()
            .map_err(|e| format!("Failed to get default keychain: {}", e))?;

        if output.status.success() {
            let keychain = String::from_utf8_lossy(&output.stdout)
                .trim()
                .trim_matches('"')
                .to_string();
            if !keychain.is_empty() {
                return Ok(keychain);
            }
        }

        // Fallback to standard login keychain location
        if let Some(home) = dirs::home_dir() {
            let login_keychain = home.join("Library/Keychains/login.keychain-db");
            if login_keychain.exists() {
                return Ok(login_keychain.to_string_lossy().to_string());
            }
        }

        Err("Could not determine login keychain path".to_string())
    }

    /// Install CA on Windows using certutil.
    #[cfg(target_os = "windows")]
    fn install_windows(ca_path: &Path) -> TrustResult {
        info!("Installing CA to Windows user certificate store");

        // Install to user store (doesn't require admin)
        let output = Command::new("certutil")
            .args(["-user", "-addstore", "Root"])
            .arg(ca_path)
            .output();

        match output {
            Ok(output) => {
                if output.status.success() {
                    info!("CA certificate installed to Windows user store");
                    TrustResult::Installed
                } else {
                    let stderr = String::from_utf8_lossy(&output.stderr);
                    let stdout = String::from_utf8_lossy(&output.stdout);

                    // Check if already exists
                    if stdout.contains("already in store") || stderr.contains("already in store") {
                        TrustResult::AlreadyInstalled
                    } else {
                        warn!("Failed to install CA: {} {}", stdout, stderr);
                        TrustResult::ManualRequired(format!(
                            "Could not auto-install CA. Run as administrator: certutil -user -addstore Root {:?}",
                            ca_path
                        ))
                    }
                }
            }
            Err(e) => TrustResult::Failed(format!("Failed to run certutil: {}", e)),
        }
    }

    /// Install CA on Linux.
    ///
    /// Linux is more complex due to distribution differences and the fact
    /// that system-wide installation typically requires root.
    #[cfg(target_os = "linux")]
    fn install_linux(ca_path: &Path) -> TrustResult {
        // Try NSS database first (works for Chrome/Chromium without root)
        if let TrustResult::Installed = Self::install_linux_nss(ca_path) {
            return TrustResult::Installed;
        }

        // Try system-wide installation (requires root, will likely fail)
        if let TrustResult::Installed = Self::install_linux_system(ca_path) {
            return TrustResult::Installed;
        }

        TrustResult::ManualRequired(
            "Linux requires manual CA installation. See instructions above.".to_string(),
        )
    }

    /// Install to NSS database (Chrome/Chromium on Linux).
    #[cfg(target_os = "linux")]
    fn install_linux_nss(ca_path: &Path) -> TrustResult {
        // Check if certutil (NSS) is available
        if Command::new("certutil").arg("--version").output().is_err() {
            debug!("NSS certutil not found, skipping NSS installation");
            return TrustResult::ManualRequired("NSS certutil not found".to_string());
        }

        let home = match dirs::home_dir() {
            Some(h) => h,
            None => return TrustResult::ManualRequired("Could not find home directory".to_string()),
        };

        let nss_db = home.join(".pki/nssdb");
        if !nss_db.exists() {
            debug!("NSS database not found at {:?}", nss_db);
            return TrustResult::ManualRequired("NSS database not found".to_string());
        }

        info!("Installing CA to NSS database at {:?}", nss_db);

        // First check if already installed
        let check_output = Command::new("certutil")
            .args(["-d", &format!("sql:{}", nss_db.display())])
            .args(["-L", "-n", "Alloy Local CA"])
            .output();

        if let Ok(output) = check_output {
            if output.status.success() {
                return TrustResult::AlreadyInstalled;
            }
        }

        // Install
        let output = Command::new("certutil")
            .args(["-d", &format!("sql:{}", nss_db.display())])
            .args(["-A", "-t", "C,,", "-n", "Alloy Local CA", "-i"])
            .arg(ca_path)
            .output();

        match output {
            Ok(output) => {
                if output.status.success() {
                    info!("CA installed to NSS database (Chrome/Chromium will trust it)");
                    TrustResult::Installed
                } else {
                    let stderr = String::from_utf8_lossy(&output.stderr);
                    debug!("NSS installation failed: {}", stderr);
                    TrustResult::ManualRequired(format!("NSS installation failed: {}", stderr))
                }
            }
            Err(e) => TrustResult::ManualRequired(format!("Failed to run certutil: {}", e)),
        }
    }

    /// Try system-wide installation on Linux (requires root).
    #[cfg(target_os = "linux")]
    fn install_linux_system(ca_path: &Path) -> TrustResult {
        // Try Debian/Ubuntu style
        let debian_dest = Path::new("/usr/local/share/ca-certificates/alloy-local-ca.crt");
        if debian_dest.parent().map(|p| p.exists()).unwrap_or(false) {
            if std::fs::copy(ca_path, debian_dest).is_ok() {
                if Command::new("update-ca-certificates").status().is_ok() {
                    info!("CA installed system-wide (Debian/Ubuntu style)");
                    return TrustResult::Installed;
                }
            }
        }

        // Try RHEL/Fedora style
        let rhel_dest = Path::new("/etc/pki/ca-trust/source/anchors/alloy-local-ca.crt");
        if rhel_dest.parent().map(|p| p.exists()).unwrap_or(false) {
            if std::fs::copy(ca_path, rhel_dest).is_ok() {
                if Command::new("update-ca-trust").status().is_ok() {
                    info!("CA installed system-wide (RHEL/Fedora style)");
                    return TrustResult::Installed;
                }
            }
        }

        TrustResult::ManualRequired("System-wide installation requires root".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trust_result_is_trusted() {
        assert!(TrustResult::Installed.is_trusted());
        assert!(TrustResult::AlreadyInstalled.is_trusted());
        assert!(!TrustResult::ManualRequired("test".to_string()).is_trusted());
        assert!(!TrustResult::Failed("test".to_string()).is_trusted());
    }

    #[test]
    fn test_install_nonexistent_file() {
        let result = TrustStore::install_ca(Path::new("/nonexistent/path/to/ca.pem"));
        assert!(matches!(result, TrustResult::Failed(_)));
    }
}
