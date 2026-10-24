import './PinkField.css'

export default function PinkField(props){
    return(
        <>
        <div className='pink-field'>
            {/* here could be different things */}
            <p className={`text-holder ${props.className}`}>{props.text}</p>
        </div>
        </>
    )
}