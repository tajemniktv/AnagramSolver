//! Immutable bytes shared by the parsers of one execution, with exact identities.
use crate::{
    contracts::{DataIdentity, DataRepresentation},
    control::Control,
};
use sha2::{Digest, Sha256};
use std::{
    fs::File,
    io::{self, Read},
    path::Path,
};

pub struct Snapshot {
    pub bytes: Vec<u8>,
    pub identity: DataIdentity,
}
impl Snapshot {
    pub fn load(path: &Path, role: &str, control: &Control) -> io::Result<Self> {
        control.check_io()?;
        let mut file = File::open(path)?;
        let mut bytes = Vec::new();
        let mut digest = Sha256::new();
        let mut buffer = [0_u8; 16 * 1024];
        loop {
            control.check_io()?;
            let n = file.read(&mut buffer)?;
            if n == 0 {
                break;
            }
            digest.update(&buffer[..n]);
            bytes.extend_from_slice(&buffer[..n]);
        }
        let identity = DataIdentity {
            role: role.to_owned(),
            representation: DataRepresentation::FileBytes,
            present: true,
            sha256: format!("{:x}", digest.finalize()),
            bytes: bytes.len() as u64,
        };
        Ok(Self { bytes, identity })
    }
    pub fn absent(role: &str) -> DataIdentity {
        DataIdentity {
            role: role.to_owned(),
            representation: DataRepresentation::FileBytes,
            present: false,
            sha256: format!("{:x}", Sha256::digest([])),
            bytes: 0,
        }
    }
}
