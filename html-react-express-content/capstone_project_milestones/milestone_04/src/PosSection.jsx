import './PosSection.css'
import PosSentence from './PosSentence.jsx'
import PosBreakdown from './PosBreakdown'

export default function PosSection(props){

    return(
        <>
        <div className='pos-sentence-section-wrapper'>
            <PosSentence></PosSentence>
            <PosBreakdown></PosBreakdown>
        </div>
        </>
    )
}