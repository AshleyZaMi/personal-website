// importing external libraries in the top
import {NavLink} from 'react-router-dom';

import './NavButton.css'

export default function(props){

    return(

        <>
        {/* change the a tag to a NavLink from react-router-dom */}
        <NavLink to={props.linkAddress}>
            
            <button id='nav-button'>
                {/* the border */}
                <div id='l-shaped-border'>
                    {/* the button name is the button text */}
                    <div id='button-name'>
                        {props.buttonName}
                    </div>
                </div>
            </button>
            
        </NavLink>
        </>
    )
}