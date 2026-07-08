const { getDefaultConfig } = require('expo/metro-config');
const path = require('path');

const config = getDefaultConfig(__dirname);

// Force Metro to ONLY resolve from mobile/node_modules
// Prevents conflicts with parent stockbot/node_modules (React 18.x vs 19.x)
config.resolver.nodeModulesPaths = [
  path.resolve(__dirname, 'node_modules'),
];
config.watchFolders = [];
config.resolver.disableHierarchicalLookup = true;

module.exports = config;
