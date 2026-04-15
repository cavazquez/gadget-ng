/// Descomposición de bloque contiguo por rango MPI (orden estable para Allgatherv).
pub fn gid_block_range(total: usize, rank: i32, size: i32) -> (usize, usize) {
    if size <= 0 {
        return (0, total);
    }
    let size = size as usize;
    let rank = rank.clamp(0, size as i32 - 1) as usize;
    let base = total / size;
    let rem = total % size;
    let lo = rank * base + rank.min(rem);
    let hi = lo + base + usize::from(rank < rem);
    (lo, hi)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blocks_cover_all() {
        let n = 10;
        let p = 3;
        let mut covered = vec![0usize; n];
        for r in 0..p {
            let (lo, hi) = gid_block_range(n, r, p);
            for x in &mut covered[lo..hi] {
                *x += 1;
            }
        }
        assert_eq!(covered, vec![1; n]);
    }
}
