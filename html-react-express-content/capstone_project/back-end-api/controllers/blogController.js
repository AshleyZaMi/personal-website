import db from '../db.js'

// get all posts
export const getPosts = async (req, res) => {
    try {
        if (!req.session.user) {
            return res.status(401).json({ message: 'Not logged in' })
        }

        // getting sql to display the posts
        const readSql =  `
        SELECT * FROM blog_posts ORDER BY created_at DESC
        `
        const result = await db.query(readSql)
        res.json(result.rows)
    } catch(err) {
        res.status(500).json({ message: err.message })
    }
}

// create post
export const createPost = async (req, res) => {
    try {
        if (!req.session.user) {
            return res.status(401).json({ message: 'Not logged in' })
        }

        const {title, content} = req.body
        const user_id = req.session.user.id

        if (!title || !content) {
            return res.status(400).json({ message: 'Title and content are required' })
        }

        // this is sql to create the blog post
        const createSql= `
            INSERT INTO blog_posts (user_id, title, content)
                VALUES ($1, $2, $3)
                RETURNING *
        `
        const userInputs = [user_id, title, content]
        const result = await db.query(createSql, userInputs)

        res.status(201).json(result.rows[0])
    } catch(err) {
        res.status(500).json({ message: err.message })
    }
}

export const getPostById = async (req, res) => {
    try {
        if (!req.session.user) {
            return res.status(401).json({ message: 'Not logged in' })
        }

        const { id } = req.params
        const result = await db.query(
            'SELECT * FROM blog_posts WHERE post_id = $1',
            [id]
        )

        if (result.rows.length === 0) {
            return res.status(404).json({ message: 'Post not found' })
        }

        res.json(result.rows[0])
    } catch(err) {
        res.status(500).json({ message: err.message })
    }
}