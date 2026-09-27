//! The fly's memory between idle sessions: the learned Kenyon-cell -> MBON efficacies,
//! the critic, where the sugar is, and a few counters, in
//! ~/.local/state/flysaver/memory.bin. Written atomically after every lesson and on
//! exit; refused on load unless it was made for this exact brain.

use std::io;
use std::path::{Path, PathBuf};

const MAGIC: &[u8; 4] = b"FLYM";
const VERSION: u32 = 1;

#[derive(Clone, Debug, PartialEq)]
pub struct Memory {
    pub efficacy: Vec<f64>,
    pub w_critic: Vec<f64>,
    pub b_critic: f64,
    pub lessons: u64,
    pub rewards: u64,
    pub blows: u64,
    /// 0 banana, 1 bread.
    pub sugar: u8,
    /// Seconds of screensaver time the sugar has been where it is.
    pub sugar_elapsed: f32,
    /// The last approach probability the mushroom body gave each fruit (banana, bread); NaN if never.
    pub last_p: [f32; 2],
}

pub fn path() -> PathBuf {
    let base = std::env::var_os("XDG_STATE_HOME").map(PathBuf::from).unwrap_or_else(|| crate::config::home().join(".local/state"));
    base.join("flysaver").join("memory.bin")
}

impl Memory {
    pub fn encode(&self, brain_fnv: u64) -> Vec<u8> {
        let mut b = Vec::with_capacity(64 + 8 * (self.efficacy.len() + self.w_critic.len()));
        b.extend_from_slice(MAGIC);
        b.extend_from_slice(&VERSION.to_le_bytes());
        b.extend_from_slice(&brain_fnv.to_le_bytes());
        b.extend_from_slice(&(self.efficacy.len() as u32).to_le_bytes());
        b.extend_from_slice(&(self.w_critic.len() as u32).to_le_bytes());
        for x in self.efficacy.iter().chain(&self.w_critic) {
            b.extend_from_slice(&x.to_le_bytes());
        }
        b.extend_from_slice(&self.b_critic.to_le_bytes());
        for c in [self.lessons, self.rewards, self.blows] {
            b.extend_from_slice(&c.to_le_bytes());
        }
        b.push(self.sugar);
        for f in [self.sugar_elapsed, self.last_p[0], self.last_p[1]] {
            b.extend_from_slice(&f.to_le_bytes());
        }
        b
    }

    /// Decode, refusing memories made for another brain or another seam.
    pub fn decode(b: &[u8], brain_fnv: u64, seam: usize, critic: usize) -> Result<Memory, String> {
        let mut i = 0;
        let mut take = |k: usize| -> Result<&[u8], String> {
            let s = b.get(i..i + k).ok_or("truncated")?;
            i += k;
            Ok(s)
        };
        if take(4)? != MAGIC {
            return Err("not a flysaver memory".into());
        }
        let u32_ = |s: &[u8]| u32::from_le_bytes(s.try_into().unwrap());
        let u64_ = |s: &[u8]| u64::from_le_bytes(s.try_into().unwrap());
        let f64_ = |s: &[u8]| f64::from_le_bytes(s.try_into().unwrap());
        let f32_ = |s: &[u8]| f32::from_le_bytes(s.try_into().unwrap());
        if u32_(take(4)?) != VERSION {
            return Err("memory from another version".into());
        }
        if u64_(take(8)?) != brain_fnv {
            return Err("memory made for another brain".into());
        }
        let (ne, nc) = (u32_(take(4)?) as usize, u32_(take(4)?) as usize);
        if (ne, nc) != (seam, critic) {
            return Err(format!("memory for a seam of {ne} and {nc} critic cells, not {seam} and {critic}"));
        }
        let efficacy = (0..ne).map(|_| take(8).map(f64_)).collect::<Result<Vec<_>, _>>()?;
        let w_critic = (0..nc).map(|_| take(8).map(f64_)).collect::<Result<Vec<_>, _>>()?;
        let b_critic = f64_(take(8)?);
        let (lessons, rewards, blows) = (u64_(take(8)?), u64_(take(8)?), u64_(take(8)?));
        let sugar = take(1)?[0].min(1);
        let (sugar_elapsed, p0, p1) = (f32_(take(4)?), f32_(take(4)?), f32_(take(4)?));
        if efficacy.iter().chain(&w_critic).any(|x| !x.is_finite()) {
            return Err("memory holds non-finite numbers".into());
        }
        Ok(Memory { efficacy, w_critic, b_critic, lessons, rewards, blows, sugar, sugar_elapsed, last_p: [p0, p1] })
    }

    pub fn save(&self, to: &Path, brain_fnv: u64) -> io::Result<()> {
        if let Some(dir) = to.parent() {
            std::fs::create_dir_all(dir)?;
        }
        // Unique temp name per process: several monitors may save at once; last rename wins.
        let tmp = to.with_extension(format!("tmp.{}", std::process::id()));
        std::fs::write(&tmp, self.encode(brain_fnv))?;
        std::fs::rename(&tmp, to)
    }

    pub fn load(from: &Path, brain_fnv: u64, seam: usize, critic: usize) -> Result<Memory, String> {
        let b = std::fs::read(from).map_err(|e| e.to_string())?;
        Memory::decode(&b, brain_fnv, seam, critic)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Memory {
        Memory {
            efficacy: vec![0.5, -1.25, 3.0],
            w_critic: vec![0.01, 0.02],
            b_critic: -0.3,
            lessons: 12,
            rewards: 5,
            blows: 1,
            sugar: 1,
            sugar_elapsed: 321.5,
            last_p: [0.56, f32::NAN],
        }
    }

    #[test]
    fn round_trips_and_refuses_foreign_memories() {
        let m = sample();
        let b = m.encode(42);
        let back = Memory::decode(&b, 42, 3, 2).unwrap();
        assert_eq!((back.efficacy, back.w_critic, back.lessons, back.sugar), (m.efficacy.clone(), m.w_critic.clone(), 12, 1));
        assert!(back.last_p[1].is_nan() && back.last_p[0] == 0.56);
        assert!(Memory::decode(&b, 43, 3, 2).unwrap_err().contains("another brain"));
        assert!(Memory::decode(&b, 42, 4, 2).is_err());
        assert!(Memory::decode(&b[..b.len() - 3], 42, 3, 2).is_err());
        assert!(Memory::decode(b"nope", 42, 3, 2).is_err());
    }

    #[test]
    fn saves_atomically_to_disk() {
        let dir = std::env::temp_dir().join(format!("flysaver-mem-{}", std::process::id()));
        let p = dir.join("memory.bin");
        sample().save(&p, 7).unwrap();
        assert_eq!(Memory::load(&p, 7, 3, 2).unwrap().blows, 1);
        assert!(std::fs::read_dir(&dir).unwrap().count() == 1, "temp file left behind");
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
