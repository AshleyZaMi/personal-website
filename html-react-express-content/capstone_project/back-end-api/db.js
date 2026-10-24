import pg from "pg"

const pool = new pg.Pool({
  host: process.env.DB_HOST,
  port: process.env.DB_PORT,
  user: process.env.DB_USER,
  password: process.env.DB_PASS,
  database: process.env.DB_NAME,
  options: `-c search_path=${process.env.DB_SCHEMA}`,
  ssl: process.env.DB_SSL.toLowerCase() == 'true' //if want to pass in a boolean but in .env we just save a string/this is a secure channel//here flag is true
})

export default pool
//1:24