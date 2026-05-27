fn main() {
    let out_dir = std::env::var("OUT_DIR").unwrap();
    let out_dir = std::path::PathBuf::from(out_dir);
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let manifest_dir = std::path::PathBuf::from(manifest_dir);
    let project_root = manifest_dir.parent().and_then(|p| p.parent()).unwrap_or(&manifest_dir);

    let candidates = [
        std::env::var("RING_PREBUILT_LIB").ok().map(std::path::PathBuf::from),
        Some(project_root.join("prebuilt/libring_core_0_17_14.a")),
        Some(project_root.join("prebuilt/libring_core.a")),
        Some(out_dir.join("libring_core_0_17_14.a")),
        Some(out_dir.join("libring_core.a")),
    ];

    let prebuilt_lib = candidates.iter().filter_map(|c| c.as_ref()).find(|p| p.exists()).cloned();

    let prebuilt_lib = match prebuilt_lib {
        Some(p) => p,
        None => {
            println!("cargo:rustc-link-search=native={}", out_dir.display());
            println!("cargo:rustc-link-lib=static=ring_core_0_17_14");
            println!("cargo:warning=No prebuilt ring library found. Run the build-ring-android GitHub Action first.");
            return;
        }
    };

    let lib_name = prebuilt_lib.file_stem().unwrap().to_str().unwrap()
        .trim_start_matches("lib");

    println!("cargo:rustc-link-search=native={}", prebuilt_lib.parent().unwrap().display());
    println!("cargo:rustc-link-lib=static={}", lib_name);
}
