const { Pool } = require("pg");
const {
  postgresHost,
  postgresUser,
  postgresPassword,
  postgresDatabase,
  connectionLimit,
} = require('../config/ApplicationSettings');


// Create a pool of clients
const pool = new Pool({
  host: postgresHost,
  user: postgresUser,
  password: postgresPassword,
  database: postgresDatabase,
  max: connectionLimit,
  idleTimeoutMillis: 30000,
  connectionTimeoutMillis: 2000,
  // ssl: {
  //   rejectUnauthorized: false, // Change to true if you want to verify server certificate
  // },
});

class Connection {
  constructor(client) {
    this.client = client;
  }

  async beginTransaction() {
    await this.client.query('BEGIN');
  }

  async commit() {
    await this.client.query('COMMIT');
  }

  async rollback() {
    await this.client.query('ROLLBACK');
  }

  async release() {
    this.client.release(); // Release the client back to the pool
  }

  async query(sql, values) {
    const result = await this.client.query(sql, values);
    return result.rows;
  }

  async queryOne(sql, values) {
    const result = await this.query(sql, values);
    return result[0];
  }

  async queryOneField(sql, values, field) {
    const result = await this.queryOne(sql, values);
    return result ? result[field] : null;
  }

  async queryCount(sql, values) {
    return await this.queryOneField(sql, values, "count");
  }
}

exports.getConnection = async function getConnection() {
  const client = await pool.connect(); // Get a client from the pool
  return new Connection(client); // Return a connection wrapped around the client
};
