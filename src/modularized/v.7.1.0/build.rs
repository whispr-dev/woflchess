extern crate winres;

fn main() {
    if cfg!(target_os = "windows") {
        let mut res = winres::WindowsResource::new();
        res.set_icon("c:/rust_projects/woflchess/icon.ico"); // Path to your .ico file
        res.compile().expect("Failed to add icon to exe");
    }
}