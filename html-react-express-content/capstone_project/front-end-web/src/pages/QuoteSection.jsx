import '../styles/QuoteSection.css'
import SectionName from '../components/SectionName.jsx'
import JapQuote from './JapQuote.jsx'
import {useEffect, useState} from 'react'

// will have a live one now so will replace to this path
const API_BASE_URL = import.meta.env.VITE_API_URL

export default function QuoteSection(props){
    const [quote, setQuote] = useState(null)

    useEffect(() => {
        const fetchQuote = async()=>  {
            try{
                const res = await fetch(API_BASE_URL + '/api/quote')
                const data = await res.json()
                setQuote(data)
            }catch(err){
                console.log("Failed to fetch quote", err)
            }
        }
         fetchQuote()
    }, [])

    return(
        <>
        <div id='quote-section'>
            <SectionName
            sectionName='Japanese quote of the day:'
            className='quote-section-name-update'
            />
            {/* Getting the Japanese Quote*/}
            {quote && (
                <JapQuote
                    JapaneseQuote={quote.japanese}
                    hiraganaReading={quote.reading}
                    EnglishTranslation={quote.meaning}
                />
            )}
            
        </div>
        </>
    )
}