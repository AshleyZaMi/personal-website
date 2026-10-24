import './Button.css'

export default function Button(props){
    return(
        <>
        <button type="submit" className={`button ${props.className}`}>
            <strong>{props.buttonText}</strong>
        </button>
        </>
    )
}