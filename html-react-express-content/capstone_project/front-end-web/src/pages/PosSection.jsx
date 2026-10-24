import '../styles/PosSection.css'
import PosSentence from './PosSentence.jsx'
import PosBreakdown from './PosBreakdown.jsx'
import { analyzePos } from './analyzePos.js'
import {useState} from 'react'

export default function PosSection(props){
    const [text, setText] = useState('')
    const [tokens, setTokens] = useState([])

    const handleAnalyze = async() => {
        console.log("clicked");
        const data = await analyzePos(text)
        // setting the tokens
        setTokens(data.tokens)
    }
    return(
        <>
        <div className='pos-sentence-section-wrapper'>
            <PosSentence
            text = {text}
            setText = {setText}
            onAnalyze = {handleAnalyze}
            />

            <PosBreakdown
            tokens = {tokens}
            />
        </div>
        </>
    )
}