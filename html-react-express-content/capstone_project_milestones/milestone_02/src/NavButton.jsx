import './NavButton.css'

export default function(props){

    return(

        <>
        <a href={props.linkAddress}>
            
            <button id='nav-button'>
                {/* the border */}
                <div id='l-shaped-border'>
                    {/* the button name is the button text */}
                    <div id='button-name'>
                        {props.buttonName}
                    </div>
                </div>
            </button>
            
        </a>
        </>
    )
}