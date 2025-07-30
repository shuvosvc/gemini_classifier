require("dotenv").config();

var ApplicationSettings = {
  port: process.env.PORT || 5000,
  jwtSecret: process.env.JWTSECRET || "trialvo",

  geminiApiKey: process.env.GEMINI_API_KEY || "AIzaSyD2Tdyug0HHt22qNbZHlaPVAwb1szSsQQE",
 // Individual PostgreSQL settings
  postgresHost: process.env.POSTGRES_HOST || "103.159.36.58",
  postgresUser: process.env.POSTGRES_USER || "trialvoc_nura",
  postgresPassword: process.env.POSTGRES_PASSWORD || "7061svcnura",
  postgresDatabase: process.env.POSTGRES_DATABASE || "trialvoc_rse",
  connectionLimit: process.env.CONNECTION_LIMIT || 100,

};

module.exports = ApplicationSettings;
