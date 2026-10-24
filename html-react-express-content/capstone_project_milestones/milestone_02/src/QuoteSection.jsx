import './QuoteSection.css'
import SectionName from './SectionName.jsx'
import JapQuote from './JapQuote.jsx'

export default function QuoteSection(props){
    return(
        <>
        <div id='quote-section'>
            <SectionName
            sectionName='Japanese quote of the day:'
            className='quote-section-name-update'
            />
            {/* Getting the Japanese Quote*/}
            <JapQuote
            JapaneseQuote="侘び寂び"
            hiraganaReading="わびさび"
            EnglishTranslation="Beauty in imperfection"
            />
            
        </div>
        </>
    )
}