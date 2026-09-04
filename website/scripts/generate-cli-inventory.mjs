import { spawnSync } from "node:child_process";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, renameSync, rmSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const websiteDir = path.resolve(scriptDir, "..");
const repoRoot = path.resolve(websiteDir, "..");
const defaultOutputPath = path.resolve(websiteDir, "src/_data/cliInventory.json");

const checkOnly = process.argv.slice(2).includes("--check");
const outputPath = defaultOutputPath;

function fail(message) {
  console.error(`CLI inventory: ${message}`);
  process.exitCode = 1;
}

function readInventory(filePath) {
  if (!existsSync(filePath)) {
    throw new Error(`generator did not create ${path.relative(repoRoot, filePath)}`);
  }

  let value;
  try {
    value = JSON.parse(readFileSync(filePath, "utf8"));
  } catch (error) {
    throw new Error(`generated file is not valid JSON: ${error.message}`);
  }

  validateInventory(value);
  return value;
}

function validateInventory(document) {
  if (!document || typeof document !== "object" || Array.isArray(document)) {
    throw new Error("generated document must be a JSON object");
  }
  if (document.schemaVersion !== 1) {
    throw new Error("generated document must declare schemaVersion 1");
  }
  if (!document.root || typeof document.root !== "object" || Array.isArray(document.root)) {
    throw new Error("generated document must contain a root object");
  }

  const paths = new Set();
  visitNode(document.root, paths, "root");
}

function visitNode(node, paths, location) {
  if (!node || typeof node !== "object" || Array.isArray(node)) {
    throw new Error(`${location} must be an object`);
  }

  for (const field of ["kind", "name", "path"]) {
    if (typeof node[field] !== "string" || node[field].length === 0) {
      throw new Error(`${location}.${field} must be a non-empty string`);
    }
  }
  if (typeof node.description !== "string") {
    throw new Error(`${location}.description must be a string`);
  }
  if (typeof node.hidden !== "boolean") {
    throw new Error(`${location}.hidden must be a boolean`);
  }
  if (!Array.isArray(node.aliases) || node.aliases.some((alias) => typeof alias !== "string")) {
    throw new Error(`${location}.aliases must be an array of strings`);
  }
  if (!Array.isArray(node.children)) {
    throw new Error(`${location}.children must be an array`);
  }
  if (!["command", "option", "positional"].includes(node.kind)) {
    throw new Error(`${location}.kind must be command, option, or positional`);
  }
  if (node.kind === "command") {
    if (typeof node.synthetic !== "boolean" || typeof node.external !== "boolean") {
      throw new Error(`${location} command nodes must declare synthetic and external booleans`);
    }
  } else {
    for (const field of ["id", "required", "global", "repeatable"]) {
      const expectedType = field === "id" ? "string" : "boolean";
      if (typeof node[field] !== expectedType) {
        throw new Error(`${location}.${field} must be a ${expectedType}`);
      }
    }
    for (const field of ["valueNames", "defaultValues", "possibleValues", "conflicts"]) {
      if (!Array.isArray(node[field]) || node[field].some((value) => typeof value !== "string")) {
        throw new Error(`${location}.${field} must be an array of strings`);
      }
    }
    if (node.children.length > 0) {
      throw new Error(`${location} leaf nodes cannot have children`);
    }
  }
  if (paths.has(node.path)) {
    throw new Error(`duplicate node path ${node.path}`);
  }
  paths.add(node.path);

  let sawLeaf = false;
  for (const [index, child] of node.children.entries()) {
    const childLocation = `${location}.children[${index}]`;
    if (child && typeof child.kind === "string") {
      const isLeaf = child.kind === "option" || child.kind === "positional";
      if (isLeaf) sawLeaf = true;
      if (!isLeaf && sawLeaf) {
        throw new Error(`${childLocation} command nodes must precede option/positional nodes`);
      }
    }
    visitNode(child, paths, childLocation);
  }
}

function runCargo(args) {
  const command = [
    "run",
    "--locked",
    "--quiet",
    "-p",
    "mesh-llm-cli",
    "--bin",
    "mesh-llm-cli-inventory",
    "--",
    ...args,
  ];
  const result = spawnSync("cargo", command, {
    cwd: repoRoot,
    env: { ...process.env },
    stdio: "inherit",
  });
  if (result.error) {
    throw new Error(`unable to run cargo: ${result.error.message}`);
  }
  if (result.status !== 0) {
    throw new Error(`inventory generator exited with status ${result.status ?? "unknown"}`);
  }
}

function generateTo(filePath) {
  runCargo([filePath]);
  readInventory(filePath);
}

function generate() {
  mkdirSync(path.dirname(outputPath), { recursive: true });
  const temporaryPath = path.join(
    path.dirname(outputPath),
    `.${path.basename(outputPath)}.${process.pid}.${Date.now()}.${os.tmpdir().replace(/[^a-z0-9]/gi, "")}.tmp`
  );
  try {
    generateTo(temporaryPath);
    renameSync(temporaryPath, outputPath);
  } finally {
    rmSync(temporaryPath, { force: true });
  }
  readInventory(outputPath);
}

function check() {
  const temporaryDirectory = mkdtempSync(path.join(os.tmpdir(), "mesh-llm-cli-inventory-"));
  const firstPath = path.join(temporaryDirectory, "first.json");
  const secondPath = path.join(temporaryDirectory, "second.json");
  try {
    generateTo(firstPath);
    runCargo(["--check", firstPath]);
    generateTo(secondPath);

    const firstBytes = readFileSync(firstPath);
    const secondBytes = readFileSync(secondPath);
    if (!firstBytes.equals(secondBytes)) {
      throw new Error("generator output is not deterministic across two runs");
    }

    if (existsSync(outputPath)) {
      readInventory(outputPath);
      runCargo(["--check", outputPath]);
      if (!firstBytes.equals(readFileSync(outputPath))) {
        throw new Error(
          `${path.relative(repoRoot, outputPath)} is stale; regenerate it from the current CLI source`
        );
      }
    }
  } finally {
    rmSync(temporaryDirectory, { force: true, recursive: true });
  }
}

try {
  if (checkOnly) {
    check();
  } else {
    generate();
  }
} catch (error) {
  fail(error instanceof Error ? error.message : String(error));
}
