fn main() {
    for d in cleanroom_video::enumerate() {
        println!(
            "{:<16} kind={:<9} virtual={:<5} accessible={:<5} driver={:<14} card={}",
            d.path.display(),
            format!("{:?}", d.kind),
            d.is_virtual,
            d.accessible,
            d.driver,
            d.card
        );
    }
    println!("\nusable inputs:");
    for d in cleanroom_video::capture_devices() {
        println!("  {} — {}", d.path.display(), d.card);
    }
}
