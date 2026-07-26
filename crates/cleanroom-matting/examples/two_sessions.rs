//! Does a second RVM session in the same process work?
//!
//! Matters because the video pipeline recreates its matter on a config change.
fn main() {
    let model = match cleanroom_matting::find_model() {
        Ok(m) => m,
        Err(e) => {
            eprintln!("no model: {e}");
            return;
        }
    };
    let px = (cleanroom_matting::INFER_W * cleanroom_matting::INFER_H) as usize;
    let frame = vec![120u8; px * 4];

    for round in 1..=2 {
        eprintln!("--- session {round} ---");
        match cleanroom_matting::Matter::new(&model) {
            Ok(mut m) => {
                match m.infer(&frame) {
                    Ok(a) => eprintln!("  inferred, matte {} bytes", a.len()),
                    Err(e) => eprintln!("  infer failed: {e}"),
                }
                eprintln!("  dropping session {round}");
            }
            Err(e) => {
                eprintln!("  could not create session {round}: {e}");
                return;
            }
        }
        eprintln!("  session {round} dropped OK");
    }
    eprintln!("BOTH SESSIONS OK");
}
