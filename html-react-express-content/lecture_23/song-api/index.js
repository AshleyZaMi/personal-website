import swaggerUi from 'swagger-ui-express' //middleware they created for us
import fs from 'fs'
import express from 'express'
import cors from 'cors'
import db from './db.js'

const toSnakeCase = str => 
  str.replace(/[A-Z]/g, letter => `_${letter.toLowerCase()}`).replace(/^_/, "");

const app = express()
const port = 3000

app.use(cors())
app.use(express.urlencoded())

// created end points for genres so don't have to do that and can load it dynamically.
//For REST API want other people if they want to use it. So there is some documentation for
// people to know how to make the documentation.
// Like know what documentation it does. 
// OPEN API -> is a specific way to do these documentations. Because some have some special ui,
// will use a package to use it
app.get("/tracks", async (req, res) => {
    try {

        const sql = `
            SELECT track_id, name, composer, milliseconds, bytes, unit_price
            FROM song_db.tracks
        `
        const results = await db.query(sql)

        res.json(results.rows.map((row) => {
            return {
                trackId: row.track_id,
                trackName: row.name,
                composer: row.composer,
                milliseconds: row.milliseconds,
                bytes: row.bytes,
                unit_price: row.unit_price
            }
        }))
    } catch (error) {
        console.log(error)
        res.status(500).json({
            message: "Error with database"
        })
    }

})

app.get("/tracks/search", async (req, res) => {
    try {
        const trackName = req.query.trackName
        const genreId = req.query.genreId
        const mediaTypeId = req.query.mediaTypeId

        let sql = `
            SELECT track_id AS "trackId", tracks.name, genres.name AS genre, media_types.name AS "mediaType"
            FROM tracks
            LEFT JOIN genres
                ON genres.genre_id = tracks.genre_id
            LEFT JOIN media_types
                ON media_types.media_type_id = tracks.media_type_id
            WHERE 1 = 1
        `

        const userInput = []
        let index = 1

        if (trackName) {
            sql += ` AND tracks.name ILIKE $${index}`
            userInput.push(`%${trackName}%`)
            index++
        }

        if (genreId) {
            sql += ` AND tracks.genre_id = $${index}`
            userInput.push(genreId)
            index++
        }

        if (mediaTypeId) {
            sql += ` AND tracks.media_type_id = $${index}`
            userInput.push(mediaTypeId)
            index++
        }

        const results = await db.query(sql, userInput)

        res.json({
            tracks: results.rows
        })

    } catch (error) {
        console.log(error)
        res.status(500).json({
            message: "Server Error"
        })
    }
})

app.get("/tracks/:trackId", async (req, res) => {
    try {
        const sql = `
            SELECT tracks.name,
                artists.name AS artist,
                composer,
                albums.title AS album,
                genres.name AS genre,
                milliseconds,
                bytes,
                unit_price AS price,
                media_types.name AS "mediaType",
                tracks.genre_id AS "genreId"
            FROM tracks
            LEFT JOIN genres
                ON genres.genre_id = tracks.genre_id
            LEFT JOIN media_types
                ON media_types.media_type_id = tracks.media_type_id
            LEFT JOIN albums
                ON albums.album_id = tracks.album_id
            LEFT JOIN artists
                ON artists.artist_id = albums.album_id
            WHERE track_id = $1
        `
        const inputValues = [req.params.trackId]
        const results = await db.query(sql, inputValues)

        res.json(results.rows[0])

    } catch (error) {
        console.log(error)
        res.status(500).json({
            message: "Server Error"
        })
    }
})

app.post("/tracks", async (req, res) => {
    try {
        const {
            trackName,
            mediaTypeId,
            genreId,
            milliseconds,
            price
        } = req.body || {}

        if (!trackName || !mediaTypeId || !genreId || !milliseconds || !price) {
            res.status(400).json({ message: "One or more required fields missing." })
            return
        }

        const sql = `
            INSERT INTO tracks (
                name,
                media_type_id,
                genre_id,
                milliseconds,
                unit_price,
                album_id,
                bytes,
                composer
            ) VALUES (
                $1,
                $2,
                $3,
                $4,
                $5,
                $6,
                $7,
                $8
            ) 
            RETURNING *
        `
        const userInput = [
            trackName,
            mediaTypeId,
            genreId,
            milliseconds,
            price
        ]

        const { albumId, bytes, composer } = req.body

        if (!albumId) {
            userInput.push(null)
        } else {
            userInput.push(albumId)
        }

        if (!bytes) {
            userInput.push(null)
        } else {
            userInput.push(bytes)
        }

        if (!composer) {
            userInput.push(null)
        } else {
            userInput.push(composer)
        }

        const results = await db.query(sql, userInput)

        const track = results.rows[0]

        res.status(201).json({
            trackId: track.track_id,
            name: track.name,
            albumId: track.album_id,
            mediaTypeId: track.media_type_id,
            genreId: track.genre_id,
            composer: track.composer,
            milliseconds: track.milliseconds,
            bytes: track.bytes,
            unitPrice: track.unit_price
        })

    } catch (error) {
        console.log(error)
        res.status(500).json({
            message: "Server Error"
        })
    }
})

app.put("/tracks/:id", async (req, res) => {
    try {
        const trackId = req.params.id

        const {
            trackName,
            mediaTypeId,
            genreId,
            milliseconds,
            price
        } = req.body || {}

        if (!trackName || !mediaTypeId || !genreId || !milliseconds || !price) {
            res.status(400).json({ message: "One or more required fields missing." })
            return
        }

        const sql = `
            UPDATE tracks
            SET name = $1,
                media_type_id = $2,
                genre_id = $3,
                milliseconds = $4,
                unit_price = $5,
                album_id = $6,
                bytes = $7,
                composer = $8
            WHERE track_id = $9
            RETURNING *
        `
        const userInput = [
            trackName,
            mediaTypeId,
            genreId,
            milliseconds,
            price
        ]

        const { albumId, bytes, composer } = req.body

        if (!albumId) {
            userInput.push(null)
        } else {
            userInput.push(albumId)
        }

        if (!bytes) {
            userInput.push(null)
        } else {
            userInput.push(bytes)
        }

        if (!composer) {
            userInput.push(null)
        } else {
            userInput.push(composer)
        }

        userInput.push(trackId)

        const results = await db.query(sql, userInput)

        const track = results.rows[0]

        res.status(200).json({
            trackId: track.track_id,
            name: track.name,
            albumId: track.album_id,
            mediaTypeId: track.media_type_id,
            genreId: track.genre_id,
            composer: track.composer,
            milliseconds: track.milliseconds,
            bytes: track.bytes,
            unitPrice: track.unit_price
        })

    } catch (error) {
        console.log(error)
        res.status(500).json({
            message: "Server Error"
        })
    }
})

app.patch("/tracks/:id", async (req, res) => {
    try {
        const trackId = req.params.id

        let sql = `
            UPDATE tracks
            SET
        `

        let index = 1;
        const userInputs = []

        for (const [key, value] of Object.entries(req.body)) {
            sql += ` ${toSnakeCase(key)} = $${index},`
            userInputs.push(value)
            index++
        }

        // removing the comma
                        //start 0, then one character before , 
        sql = sql.slice(0, -1)
        sql += ` WHERE track_id = $${index}
            RETURNING *`
        userInputs.push(trackId)

        const results = await db.query(sql, userInputs)

        const track = results.rows[0]

        // can instead of this do toCamelCase if we wanted to for mapping
        res.status(200).json({
            trackId: track.track_id,
            name: track.name,
            albumId: track.album_id,
            mediaTypeId: track.media_type_id,
            genreId: track.genre_id,
            composer: track.composer,
            milliseconds: track.milliseconds,
            bytes: track.bytes,
            unitPrice: track.unit_price
        })

    } catch (error) {
        console.log(error)
        res.status(500).json({
            message: "Server Error"
        })
    }
})

app.delete("/tracks/:id", async (req, res) => {
    try {
        const trackId = req.params.id

        const sql = `
            DELETE FROM tracks
            WHERE track_id = $1
            RETURNING *
        `
        const userInput = [trackId]

        const results = await db.query(sql, userInput)

        const track = results.rows[0]

        if (track) {
            res.json({
                trackId: track.track_id,
                name: track.name,
                albumId: track.album_id,
                mediaTypeId: track.media_type_id,
                genreId: track.genre_id,
                composer: track.composer,
                milliseconds: track.milliseconds,
                bytes: track.bytes,
                unitPrice: track.unit_price
            })
        } else {
            res.status(404).json({
                message: "Resource does not exist."
            })
        }


    } catch (error) {
        console.log(error)
        res.status(500).json({
            message: "Server Error"
        })
    }
})

app.get("/genres", async (req, res) => {
    try {

        const sql = `
            SELECT genre_id AS "genreId", name
            FROM genres
        `
        const results = await db.query(sql)

        res.json(results.rows)
    } catch (error) {
        console.log(error)
        res.status(500).json({
            message: "Error with database"
        })
    }

})

app.get("/mediaTypes", async (req, res) => {
    try {

        const sql = `
            SELECT media_type_id AS "mediaTypeId", name
            FROM media_types
        `
        const results = await db.query(sql)

        res.json(results.rows)
    } catch (error) {
        console.log(error)
        res.status(500).json({
            message: "Error with database"
        })
    }

})

app.get("/albums", async (req, res) => {
    try {

        const sql = `
            SELECT album_id AS "albumId", title
            FROM albums
        `
        const results = await db.query(sql)

        res.json(results.rows)
    } catch (error) {
        console.log(error)
        res.status(500).json({
            message: "Error with database"
        })
    }

})

// will create an endpoint which will server the swaggereditor like 

// have to tell node that its not a module__

// long way of reading our json file for our application
const swaggerDocument = JSON.parse(
    // file system will allow us to write to file or read to file
    // read file synchronously
                //open this file -> the one where we have our documentation, then our path
    fs.readFileSync(new URL('./openapi.json', import.meta.url))
)

// right now-> right now is static so if we do make it static will have do npm run swagger in order to update the
// changes that have regenerate the documenation. Just like for react if we make a change we have to rebuild the code.

// since we created route can use use, so we can see the ui of our api documentation
                    // want to serve it, and then want to pass the document and create all the ui compnents()
// app.use('/api-docs', swaggerUi.serve, swaggerUi.setup(swaggerDocument))

// making it an array just incase users want to see it in homepage
// app.use(['','/api-docs'], swaggerUi.serve, swaggerUi.setup(swaggerDocument))
app.use('', swaggerUi.serve, swaggerUi.setup(swaggerDocument))
app.use('/api-docs', swaggerUi.serve, swaggerUi.setup(swaggerDocument))

app.listen(port, () => {
    console.log(`Server running on http://localhost:${port}`)
})
