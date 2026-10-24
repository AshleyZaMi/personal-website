// importing external libraries in the top
import {NavLink} from 'react-router-dom';

import NavButton from './NavButton.jsx'

import './Navbar.css'

export default function Navbar(props){
    return(
        // this is also a parent container of children
        // logo, tabs-section
        <div id="navbar-wrapper">
            <div id="logo">
                {/* change a tags to NavLink */}
                <NavLink to="/">ようこそ！</NavLink>
            </div> {/* #logo ends */}
            {/* want to use section many times, and also same design*/}
            {/*this is a parent of children using section class*/}
            <div id='tabs-section'>
                <div className="section">
                    <NavButton
                    linkAddress="/blog"
                    buttonName='Blog'
                    >
                    </NavButton>
                    
                    
                </div>{/* .section ends */}
                <div className="section">
                    <NavButton
                    linkAddress="/analysis"
                    buttonName='Analysis'>
                    </NavButton>
                </div> {/* .section ends */}
            </div>{/* #tabs-section */}
            
        </div>
    )
}