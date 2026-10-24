import '../styles/PosSentence.css'
import SectionName from '../components/SectionName.jsx'
import PinkField from '../components/PinkField.jsx'
import Button from '../components/Button.jsx'

export default function PosSentence(props){
    return(
        <>
        <SectionName
        sectionName='Enter sentence'
        ></SectionName>
        <PinkField pinkStyling='pos-pink-text-styling'>
            {/* inside PinkField are children */}
            <label htmlFor='new-pos-sentence'></label>
            <input 
                type='text' 
                id='new-pos-sentence'
                value={props.text}
                // on change will set the Text to the new value pm the PosSection(parent of this section)
                onChange={(event) => props.setText(event.target.value)}
                >
            </input>
        </PinkField>

        <div id='pos-button-wrapper'>
            <Button
                className='pos-button-styling'
                buttonText='POS it'
                onClick={props.onAnalyze}
            ></Button>
        </div>
        </>
    )
}