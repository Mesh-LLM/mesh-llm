// Generate macOS .icns from the jellyfish icon-512.png
// Usage: swift gen_icon.swift

import AppKit

let srcPath = "../mesh-llm/ui/public/icon-512.png"
let iconsetDir = "MeshLLM.app/Contents/Resources/AppIcon.iconset"

guard let srcImage = NSImage(contentsOfFile: srcPath) else {
    print("❌ Could not load \(srcPath)")
    exit(1)
}

let fm = FileManager.default
try? fm.removeItem(atPath: iconsetDir)
try! fm.createDirectory(atPath: iconsetDir, withIntermediateDirectories: true)

let sizes: [(String, Int)] = [
    ("icon_16x16.png", 16),
    ("icon_16x16@2x.png", 32),
    ("icon_32x32.png", 32),
    ("icon_32x32@2x.png", 64),
    ("icon_128x128.png", 128),
    ("icon_128x128@2x.png", 256),
    ("icon_256x256.png", 256),
    ("icon_256x256@2x.png", 512),
    ("icon_512x512.png", 512),
    ("icon_512x512@2x.png", 1024),
]

for (name, px) in sizes {
    let rep = NSBitmapImageRep(
        bitmapDataPlanes: nil,
        pixelsWide: px, pixelsHigh: px,
        bitsPerSample: 8, samplesPerPixel: 4,
        hasAlpha: true, isPlanar: false,
        colorSpaceName: .deviceRGB,
        bytesPerRow: 0, bitsPerPixel: 0
    )!
    rep.size = NSSize(width: px, height: px)

    NSGraphicsContext.saveGraphicsState()
    NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: rep)
    srcImage.draw(in: NSRect(x: 0, y: 0, width: px, height: px))
    NSGraphicsContext.restoreGraphicsState()

    let data = rep.representation(using: .png, properties: [:])!
    let path = (iconsetDir as NSString).appendingPathComponent(name)
    try! data.write(to: URL(fileURLWithPath: path))
}

print("✅ Generated iconset at \(iconsetDir)")
print("   Run: iconutil -c icns \(iconsetDir) -o MeshLLM.app/Contents/Resources/AppIcon.icns")
