import pg from "pg"

const pool = new pg.Pool({
  host: process.env.DB_HOST,
  port: process.env.DB_PORT,
  user: process.env.DB_USER,
  password: process.env.DB_PASS,
  database: process.env.DB_NAME,
  // updating this
  // if it is false then will be all or if false will be false
  // save a boolean with the env. The secured socket, which will prevent
  // man in the middle only want databse to connect without application,
  // and don't want people to intercept those connections,
  // so locally it will be false, but online will not be false, so want to create
  // secure chanel. So on web will be true, but locally will be false.
  ssl:process.env.DB_SSL.toLowerCase() == 'true'
})

pool.on('connect', async (client) => {
  await client.query(`SET search_path TO ${process.env.DB_SCHEMA}`)
})

export default pool