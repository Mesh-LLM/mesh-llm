import { readFileSync } from "node:fs";

const publishScript = readFileSync(new URL("../../../scripts/publish-crates.sh", import.meta.url), "utf8");
const publishList = publishScript.match(/publish_crates=\(([\s\S]*?)\n\)/)?.[1];

if (!publishList) {
  throw new Error("Unable to determine the published crate list");
}

const crateNames = Array.from(publishList.matchAll(/^    ([a-z0-9_-]+)$/gm), ([, name]) => name);

export default crateNames.map((name) => ({
  name,
  rustdocName: name.replaceAll("-", "_"),
}));
