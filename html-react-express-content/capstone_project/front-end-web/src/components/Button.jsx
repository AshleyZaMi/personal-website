import '../styles/Button.css'

export default function Button(props){
    return(
        <button 
            type="button" 
            className={`button ${props.className}`}
            onClick={props.onClick}>
            <strong>{props.buttonText}</strong>
        </button>
    )
}