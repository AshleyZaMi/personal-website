import express from 'express'

import {
    createSession
} from '../controllers/pomodoroController.js'

const router = express.Router()

router.post('/sessions', createSession)

export default router