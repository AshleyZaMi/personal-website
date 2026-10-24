import '../styles/WelcomeText.css'

export default function WelcomeText(props){
    const text = props.write

    return(
        <h1 className='welcome-text'>
            {text.split("").map((char, index) => (
                <span
                key={index}
                className='welcome-letter'
                style={{ animationDelay: `${index * 0.15}s`}}
                >
                    {/* this is the body showing it, showing the letter */}
                    {char}
                </span>
            ))}
        </h1>
    )
}