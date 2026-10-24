import '../styles/PinkField.css'

export default function PinkField(props){
    return(
        <>
        <div className={`pink-field ${props.pinkStyling}`}>
            {/* here could be different things */}
            <p className={`text-holder ${props.className}`}>{props.text}</p>

            {/* will show any children(ex. if i were to make it an input field) */}
            {props.children}
        </div>
        </>
    )
}