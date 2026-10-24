// importing external libraries in the top
import {NavLink, useNavigate} from 'react-router-dom';
import NavButton from './NavButton.jsx'

import '../styles/Navbar.css'

// will have a live one now so will replace to this path
const API_BASE_URL = import.meta.env.VITE_API_URL

export default function Navbar(props){
    // using navigate to redirect the users after they log out
    const navigate = useNavigate()

    const handleLogout = async() => {
        await fetch(API_BASE_URL + '/logout', {
            method: 'GET',
            credentials: 'include'
        })

        // making them go back to login page
        navigate('/login')
    }
    return(
        // this is also a parent container of children
        // logo, tabs-section
        <div id="navbar-wrapper">
            <div id="logo">
                {/* change a tags to NavLink */}
                <NavLink to="/home">ようこそ！</NavLink>
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

            <div id='logout-button-wrapper'>
                <button 
                className='btn btn-outline-danger px-4 py-3'
                onClick={handleLogout}
                >
                    Logout
                </button>
            </div>

            
        </div>
    )
}