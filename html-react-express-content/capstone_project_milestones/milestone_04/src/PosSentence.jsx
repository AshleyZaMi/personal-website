import './PosSentence.css'
import SectionName from './SectionName.jsx'
import PinkField from './PinkField.jsx'
import Button from './Button.jsx'

export default function PosSentence(props){
    return(
        <>
        <SectionName
        sectionName='Enter sentence'
        ></SectionName>
        <PinkField
        pinkStyling='pos-pink-text-styling'
        text = {props.userGivenText}
        >
            {/* inside PinkField are children */}
            <label htmlFor='new-pos-sentence'></label>
            <input type='text' id='new-pos-sentence'></input>
        </PinkField>
        <div id='pos-button-wrapper'>
            <Button
            className='pos-button-styling'
            buttonText='POS it'
            ></Button>
        </div>
        </>
    )
}