//! Certificate generation and management.
//!
//! Creates a local Certificate Authority (CA) and generates certificates
//! signed by that CA for localhost development.

use std::fs;
use std::net::IpAddr;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use rcgen::{
    BasicConstraints, CertificateParams, DnType, ExtendedKeyUsagePurpose, IsCa, KeyPair,
    KeyUsagePurpose, SanType, Certificate,
};
use tracing::{debug, info};

use super::trust_store::{TrustResult, TrustStore};

/// Generated certificate and key pair.
#[derive(Debug, Clone)]
pub struct GeneratedCerts {
    /// PEM-encoded certificate
    pub cert_pem: String,
    /// PEM-encoded private key
    pub key_pem: String,
}

/// Manages local CA and certificate generation.
pub struct CertManager {
    /// Directory where certificates are stored
    certs_dir: PathBuf,
    /// CA certificate PEM
    ca_cert_pem: String,
    /// CA certificate (for signing)
    ca_cert: Certificate,
    /// CA private key
    ca_key_pair: KeyPair,
}

impl CertManager {
    /// Create a new certificate manager, loading or generating the CA.
    ///
    /// The CA is stored in `{data_dir}/certs/` and persists across restarts.
    pub fn new(data_dir: &Path) -> Result<Self> {
        let certs_dir = data_dir.join("certs");
        fs::create_dir_all(&certs_dir)
            .with_context(|| format!("Failed to create certs directory: {:?}", certs_dir))?;

        let ca_cert_path = certs_dir.join("alloy-ca.pem");
        let ca_key_path = certs_dir.join("alloy-ca-key.pem");

        let (ca_cert_pem, ca_cert, ca_key_pair) = if ca_cert_path.exists() && ca_key_path.exists() {
            debug!("Loading existing CA from {:?}", certs_dir);
            Self::load_ca(&ca_cert_path, &ca_key_path)?
        } else {
            info!("Generating new local CA certificate");
            let (cert_pem, cert, key_pair) = Self::generate_ca()?;
            Self::save_ca(&cert_pem, &key_pair, &ca_cert_path, &ca_key_path)?;
            (cert_pem, cert, key_pair)
        };

        Ok(Self {
            certs_dir,
            ca_cert_pem,
            ca_cert,
            ca_key_pair,
        })
    }

    /// Create CA certificate parameters.
    fn create_ca_params() -> CertificateParams {
        let mut params = CertificateParams::default();

        // Set as CA
        params.is_ca = IsCa::Ca(BasicConstraints::Unconstrained);

        // Set distinguished name
        params
            .distinguished_name
            .push(DnType::CommonName, "Alloy Local Development CA");
        params
            .distinguished_name
            .push(DnType::OrganizationName, "Alloy");
        params
            .distinguished_name
            .push(DnType::OrganizationalUnitName, "Development");

        // CA should be valid for 10 years
        params.not_before = time::OffsetDateTime::now_utc();
        params.not_after = time::OffsetDateTime::now_utc() + time::Duration::days(3650);

        // Key usage for CA
        params.key_usages = vec![KeyUsagePurpose::KeyCertSign, KeyUsagePurpose::CrlSign];

        params
    }

    /// Generate a new Certificate Authority.
    fn generate_ca() -> Result<(String, Certificate, KeyPair)> {
        let params = Self::create_ca_params();

        // Generate key pair
        let key_pair = KeyPair::generate().context("Failed to generate CA key pair")?;

        // Self-sign the CA certificate
        let cert = params
            .self_signed(&key_pair)
            .context("Failed to self-sign CA certificate")?;

        let cert_pem = cert.pem();

        Ok((cert_pem, cert, key_pair))
    }

    /// Load existing CA from disk.
    ///
    /// Since we can't deserialize the Certificate struct, we regenerate it
    /// from the stored key pair using the same parameters.
    fn load_ca(cert_path: &Path, key_path: &Path) -> Result<(String, Certificate, KeyPair)> {
        let cert_pem =
            fs::read_to_string(cert_path).context("Failed to read CA certificate file")?;
        let key_pem = fs::read_to_string(key_path).context("Failed to read CA key file")?;

        let key_pair =
            KeyPair::from_pem(&key_pem).context("Failed to parse CA private key from PEM")?;

        // Regenerate the CA certificate with the loaded key pair
        // This ensures we have a valid Certificate object for signing
        let params = Self::create_ca_params();
        let cert = params
            .self_signed(&key_pair)
            .context("Failed to regenerate CA certificate from key")?;

        Ok((cert_pem, cert, key_pair))
    }

    /// Save CA to disk.
    fn save_ca(
        cert_pem: &str,
        key_pair: &KeyPair,
        cert_path: &Path,
        key_path: &Path,
    ) -> Result<()> {
        fs::write(cert_path, cert_pem).context("Failed to write CA certificate")?;
        fs::write(key_path, key_pair.serialize_pem()).context("Failed to write CA key")?;

        // Set restrictive permissions on key file (Unix only)
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = fs::metadata(key_path)?.permissions();
            perms.set_mode(0o600);
            fs::set_permissions(key_path, perms)?;
        }

        info!("CA certificate saved to {:?}", cert_path);
        Ok(())
    }

    /// Generate a certificate for localhost signed by the local CA.
    pub fn generate_localhost_cert(&self) -> Result<GeneratedCerts> {
        let mut params = CertificateParams::default();

        // Distinguished name
        params
            .distinguished_name
            .push(DnType::CommonName, "localhost");
        params
            .distinguished_name
            .push(DnType::OrganizationName, "Alloy");

        // Subject Alternative Names (what the cert is valid for)
        params.subject_alt_names = vec![
            SanType::DnsName("localhost".try_into()?),
            SanType::IpAddress(IpAddr::V4([127, 0, 0, 1].into())),
            SanType::IpAddress(IpAddr::V6([0, 0, 0, 0, 0, 0, 0, 1].into())),
        ];

        // Valid for 1 year
        params.not_before = time::OffsetDateTime::now_utc();
        params.not_after = time::OffsetDateTime::now_utc() + time::Duration::days(365);

        // Key usage for server certificate
        params.key_usages = vec![
            KeyUsagePurpose::DigitalSignature,
            KeyUsagePurpose::KeyEncipherment,
        ];
        params.extended_key_usages = vec![ExtendedKeyUsagePurpose::ServerAuth];

        // Generate key pair for this certificate
        let key_pair = KeyPair::generate().context("Failed to generate localhost key pair")?;

        // Sign with CA
        let cert = params
            .signed_by(&key_pair, &self.ca_cert, &self.ca_key_pair)
            .context("Failed to sign localhost certificate with CA")?;

        debug!("Generated localhost certificate");

        Ok(GeneratedCerts {
            cert_pem: cert.pem(),
            key_pem: key_pair.serialize_pem(),
        })
    }

    /// Get the path to the CA certificate file.
    pub fn ca_cert_path(&self) -> PathBuf {
        self.certs_dir.join("alloy-ca.pem")
    }

    /// Get the CA certificate PEM.
    pub fn ca_cert_pem(&self) -> &str {
        &self.ca_cert_pem
    }

    /// Attempt to install the CA certificate into the system trust store.
    ///
    /// Returns information about whether installation succeeded and any
    /// manual steps needed.
    pub fn install_ca_to_trust_store(&self) -> TrustResult {
        let ca_path = self.ca_cert_path();
        TrustStore::install_ca(&ca_path)
    }

    /// Print instructions for manually trusting the CA certificate.
    pub fn print_trust_instructions(&self) {
        let ca_path = self.ca_cert_path();
        let ca_path_str = ca_path.display();

        eprintln!();
        eprintln!("To trust the Alloy CA certificate, run one of the following:");
        eprintln!();

        #[cfg(target_os = "macos")]
        {
            eprintln!("  # macOS (adds to login keychain):");
            eprintln!(
                "  security add-trusted-cert -r trustRoot -k ~/Library/Keychains/login.keychain-db {}",
                ca_path_str
            );
            eprintln!();
        }

        #[cfg(target_os = "linux")]
        {
            eprintln!("  # Debian/Ubuntu:");
            eprintln!(
                "  sudo cp {} /usr/local/share/ca-certificates/alloy-ca.crt",
                ca_path_str
            );
            eprintln!("  sudo update-ca-certificates");
            eprintln!();
            eprintln!("  # RHEL/Fedora:");
            eprintln!(
                "  sudo cp {} /etc/pki/ca-trust/source/anchors/alloy-ca.crt",
                ca_path_str
            );
            eprintln!("  sudo update-ca-trust");
            eprintln!();
            eprintln!("  # Chrome/Chromium (user-level):");
            eprintln!(
                "  certutil -d sql:$HOME/.pki/nssdb -A -t \"C,,\" -n \"Alloy Local CA\" -i {}",
                ca_path_str
            );
            eprintln!();
        }

        #[cfg(target_os = "windows")]
        {
            eprintln!("  # Windows (run as Administrator for system-wide, or without for user):");
            eprintln!("  certutil -user -addstore Root {}", ca_path_str);
            eprintln!();
        }

        eprintln!("  CA certificate location: {}", ca_path_str);
        eprintln!();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_generate_ca() {
        let (cert_pem, _cert, _key_pair) = CertManager::generate_ca().unwrap();
        assert!(cert_pem.contains("BEGIN CERTIFICATE"));
        assert!(cert_pem.contains("END CERTIFICATE"));
    }

    #[test]
    fn test_cert_manager_new() {
        let temp_dir = TempDir::new().unwrap();
        let manager = CertManager::new(temp_dir.path()).unwrap();

        // CA files should be created
        assert!(manager.ca_cert_path().exists());
        assert!(temp_dir.path().join("certs/alloy-ca-key.pem").exists());
    }

    #[test]
    fn test_generate_localhost_cert() {
        let temp_dir = TempDir::new().unwrap();
        let manager = CertManager::new(temp_dir.path()).unwrap();

        let certs = manager.generate_localhost_cert().unwrap();
        assert!(certs.cert_pem.contains("BEGIN CERTIFICATE"));
        assert!(certs.key_pem.contains("BEGIN PRIVATE KEY"));
    }

    #[test]
    fn test_reload_existing_ca() {
        let temp_dir = TempDir::new().unwrap();

        // Create initial manager (generates CA)
        let manager1 = CertManager::new(temp_dir.path()).unwrap();
        let ca_pem1 = manager1.ca_cert_pem().to_string();

        // Create second manager (should load existing CA)
        let manager2 = CertManager::new(temp_dir.path()).unwrap();
        let ca_pem2 = manager2.ca_cert_pem().to_string();

        // Should be the same CA (the stored PEM, not regenerated)
        assert_eq!(ca_pem1, ca_pem2);
    }
}
