const pairs = [
  ["#212121", "#FFFFFF", 4.5, "body on panel"],
  ["#323A45", "#FFFFFF", 4.5, "secondary heading on panel"],
  ["#105BD8", "#FFFFFF", 4.5, "action blue text on white"],
  ["#FFFFFF", "#105BD8", 4.5, "button text on action blue"],
  ["#FFFFFF", "#15803D", 4.5, "PASS badge"],
  ["#FFFFFF", "#92400E", 4.5, "HOLD badge"],
  ["#FFFFFF", "#B91C1C", 4.5, "FAIL badge"]
];

function hexToRgb(hex) {
  const value = hex.replace("#", "");
  return [
    parseInt(value.slice(0, 2), 16),
    parseInt(value.slice(2, 4), 16),
    parseInt(value.slice(4, 6), 16)
  ].map((channel) => channel / 255);
}

function linearize(value) {
  return value <= 0.03928 ? value / 12.92 : Math.pow((value + 0.055) / 1.055, 2.4);
}

function luminance(hex) {
  const [r, g, b] = hexToRgb(hex).map(linearize);
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

function contrast(foreground, background) {
  const first = luminance(foreground);
  const second = luminance(background);
  const [lighter, darker] = first > second ? [first, second] : [second, first];
  return (lighter + 0.05) / (darker + 0.05);
}

let failed = false;

for (const [foreground, background, minimum, label] of pairs) {
  const value = contrast(foreground, background);
  if (value < minimum) {
    console.error(`[FAIL] ${label}: ${value.toFixed(3)} < ${minimum}`);
    failed = true;
  } else {
    console.log(`[PASS] ${label}: ${value.toFixed(3)}`);
  }
}

if (failed) process.exit(1);
console.log("[PASS] Quali contrast check complete");
