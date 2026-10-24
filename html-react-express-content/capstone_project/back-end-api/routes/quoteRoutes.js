import express from 'express'
import{quote} from '../controllers/quoteController.js'

const router = express.Router()

// route for random quote
router.get('/quote', quote)

export default router