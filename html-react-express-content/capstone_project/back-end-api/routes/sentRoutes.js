import express from 'express'
import{sentence} from '../controllers/sentController.js'

const router = express.Router()

// route for random topic
router.get('/sentence', sentence)

export default router