const jwt = require('jsonwebtoken');

const { jwtSecret } = require('../config/ApplicationSettings');


async function authintication(token,member_id, connection) {
  if (!token) {
    throw new Error('Access token is required.');
  }

  let decodedToken;
  try {
    decodedToken = jwt.verify(token, jwtSecret);
  } catch (error) {
    throw new Error('Invalid or expired access token.');
  }

  // Check if user exists in the database

  const isExist = await connection.queryOne(
    'SELECT user_id,profile_image_url FROM users WHERE  user_id = $1 and mc_id = $2 and deleted=false',
    [member_id, decodedToken.userId]
  );

  if (!isExist || !isExist.user_id) {
    throw new Error('Invalid user.');
  }

  return {decodedToken, isExist  };
}


const authfilereq = (req, res, next) => {
  const token = req.headers['authorization'];

  if (!token) {
    return res.status(403).json({ error: 'Access token is required.' });
  }

  jwt.verify(token, jwtSecret, (err, decoded) => {
    if (err) {
      return res.status(403).json({ error: 'Invalid or expired access token.' });
    }

    req.user = decoded; // Attach user data to the request

    // Validate file ownership
    const requestedFile = req.path.split('/').pop(); // Extract the filename from the request path
    const userIdFromFile = requestedFile.split('-').slice(-2, -1)[0]; // Extract the user ID from the filename
    
    if (userIdFromFile !== req.user.userId.toString()) {
      return res.status(403).json({ error: 'Unauthorized access to this file.' });
    }

    next(); // Proceed to the next middleware or route handler
  });
};


module.exports = { authintication ,authfilereq};
