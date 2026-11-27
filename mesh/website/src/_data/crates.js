import { execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";

const rosterScript = fileURLToPath(new URL("../../../../scripts/published-crates.sh", import.meta.url));
const crateNames = execFileSync(rosterScript, { encoding: "utf8" }).trim().split("\n");
if (crateNames.length === 0 || crateNames.some((name) => !/^[a-z0-9_-]+$/.test(name))) {
  throw new Error("Published crate roster is empty or invalid");
}

export default crateNames.map((name) => ({
  name,
  rustdocName: name.replaceAll("-", "_"),
}));
