/**
 * Wrapper for mcp-server-google-calendar to dynamically mock CLI arguments
 * and start the server correctly inside the container.
 */
import path from 'path';
import { fileURLToPath } from 'url';
import { dirname } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Resolve the package path dynamically from the original src/agent node_modules
const packageIndex = path.resolve(__dirname, 'node_modules', 'mcp-server-google-calendar', 'build', 'index.js');

// Mock command line arguments so the package thinks it's being called with the 'run' command.
// Otherwise it exits with "Usage: npx mcp-server-google-calendar@latest init" because it parses process.argv.
process.argv = [process.argv[0], packageIndex, 'run'];

// Import and run the server
await import(`file://${packageIndex}`);
