import '../styles/RecommenderSection.css'
import SectionName from '../components/SectionName.jsx'
import PinkField from '../components/PinkField.jsx'
import Button from '../components/Button.jsx'

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
                    // onNext is called in the sentenceRecommender.jsx
                    onClick={props.onNext}
                ></Button>
            </div>

        </div>
        </>
    )
}