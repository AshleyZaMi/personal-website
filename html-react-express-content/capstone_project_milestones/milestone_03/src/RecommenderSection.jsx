import './RecommenderSection.css'
import SectionName from './SectionName.jsx'
import PinkField from './PinkField.jsx'
import Button from './Button'

export default function RecommenderSection(props){

    return(
        <>
        <div id='recommender-wrapper'>
            <SectionName
            sectionName={props.sectionName}
            ></SectionName>
            <PinkField
            pinkStyling = 'recc-pink-field-styling'
            // this will be the text from the datafield right now will be hardcoded
            text ={props.databaseText}
            >
            </PinkField>
            <div id="button-centered">
                <Button
                className='button-extra-styling'
                buttonText = 'Next'
                ></Button>
            </div>

        </div>
        </>
    )
}