import RecommenderSection from './RecommenderSection.jsx'
import {useState} from 'react'

// will have a live one now so will replace to this path
const API_BASE_URL = import.meta.env.VITE_API_URL

export default function TopicRecSection(props){
    const [sentence, setSentence] = useState('Click next to load sentence')

    async function handleNext(){
        try{
            const response = await fetch(API_BASE_URL + '/api/sentence')
            const data = await response.json();
            setSentence(data.sentence)
        }catch(err){
            console.error('failed to fetch sentence', err)
        }
    }
    return(
        <>
        <RecommenderSection
        sectionName='Topic recommender'
        databaseText = {sentence}
        onNext={handleNext}
        >
        </RecommenderSection>
        </>
    )
}