import express from 'express'
import { 
    getPosts, 
    createPost,
    getPostById
} from '../controllers/blogController.js'

const router = express.Router()

router.get('/posts', getPosts)
router.post('/posts', createPost)
// this one is to be able to view the individual blogs
router.get('/posts/:id', getPostById)

export default router