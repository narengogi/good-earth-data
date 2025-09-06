
const http = require('http');
const fs = require('fs');
const path = require('path');

const PORT = 8787;

const server = http.createServer((req, res) => {
  // Parse the URL to determine the file path
  let filePath = '.' + req.url;
  if (filePath === './') {
    filePath = './index.html'; // Serve index.html by default
  }

  // Get the content type based on the file extension
  const extname = String(path.extname(filePath)).toLowerCase();
  const mimeTypes = {
    '.html': 'text/html',
    '.js': 'text/javascript',
    '.css': 'text/css',
    '.json': 'application/json',
    '.png': 'image/png',
    '.jpg': 'image/jpg',
    '.gif': 'image/gif',
    '.svg': 'image/svg+xml',
    '.wav': 'audio/wav',
    '.mp4': 'video/mp4',
    '.woff': 'application/font-woff',
    '.ttf': 'application/font-ttf',
    '.eot': 'application/vnd.ms-fontobject',
    '.otf': 'application/font-otf',
    '.ico': 'image/x-icon',
    '.pbf': 'application/x-protobuf'
  };

  const contentType = mimeTypes[extname] || 'application/octet-stream';

  // Read the file and serve it
  fs.readFile(filePath, (error, content) => {
    if (error) {
      if (error.code == 'ENOENT') {
        // File not found
        res.writeHead(404, { 'Content-Type': 'text/html' });
        res.end('<h1>404 Not Found</h1>', 'utf-8');
      } else {
        // Server error
        res.writeHead(500);
        res.end('Sorry, check with the site admin for error: ' + error.code + ' ..\n');
        res.end();
      }
    } else {
      // Success
      res.writeHead(200, {
        'Content-Type': contentType,
        'Access-Control-Allow-Origin': '*', // Or whatever your frontend port is
      });
      res.end(content, 'utf-8');
    }
  });
});

server.listen(PORT, () => {
  console.log(`Server running at http://localhost:${PORT}/`);
}); 