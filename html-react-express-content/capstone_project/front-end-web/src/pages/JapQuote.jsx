import '../styles/JapQuote.css'

export default function JapQuote(props){
    return(
        <>
        {/* Getting the Japanese Quote*/}
        <p id='jap-quote'>"{props.JapaneseQuote}" 
            <span id="jap-quote-reading"> ({props.hiraganaReading})</span>
        </p>
        {/* Getting the English Translation */}
        <p id='eng-translation'><strong>"{props.EnglishTranslation}"</strong></p>

        </>
    )
}