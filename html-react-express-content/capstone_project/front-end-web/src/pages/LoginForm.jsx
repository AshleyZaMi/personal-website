import {useState} from 'react'
import {useNavigate, Link} from 'react-router-dom'
import '../styles/LoginForm.css'

// will have a live one now so will replace to this path
const API_BASE_URL = import.meta.env.VITE_API_URL

export default function LoginPage(props){
    const [username, setUsername] = useState('')
    const [password, setPassword] = useState('')
    // taking care of errors here
    const [error, setError] = useState('')

    const navigate = useNavigate()

    const handleSubmit = async(event) =>{
        // stopping page from refreshing
        event.preventDefault()

        // here intially setting to empty string
        setError('')

        // adding validation before fetch
        if(!username.trim() || !password.trim()){
            setError('Enter username or password')
            return;
        }
        try{
            const response = await fetch(API_BASE_URL + "/login", {
                // same thing as postman but making sure to set method to POST
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                // URLSearhParams will turn it into like -> username=ashley&password=123
                body: JSON.stringify({
                    // sending the username and password
                    username,
                    password
                }),
                credentials: "include"
            })

            // the result is going to be the body, lets read the response
            const result = await response.json()

            // checking if login failed
            if(!response.ok){
                setError(result.message)
                return;
            }

            // if login is successful then will send them to the homepage
            navigate('/home')
        }catch(error){
            console.log(error)
            setError('Cannot connect to server')
        }
    }
    return(
        <div className='bootstrap-form-wrapper'>
        <h2>Login</h2>
        {/* // using bootstrap for the form */}
            <form onSubmit={handleSubmit}>
                {/* handling the case of error first this is the error message */}
                <div className='row mb-3'>
                    <div className='font-italic text-danger col-sm-9 ml-sm-auto'>
                        {error && error}
                    </div>
                </div>

                {/* form group row specifically for username*/}
                <div className='form-group row'>
                    {/* col-form-;abe; is so that they are vertically centered with their associated controls */}
                    <label htmlFor="inputUsername" className='col-sm-2 col-form-label'>
                        Username
                    </label>
                    {/* getting the input */}
                    <div className='col-sm-10'>
                        {/* getting the user input */}
                        <input 
                        // id is to locate which input field the label corresponds to
                        id="inputUsername"
                        className='form-control' //for nice display of input field
                        value={username}
                        onChange= {(event) => setUsername(event.target.value)}
                        />
                    </div>
                </div>

                {/* another form row for password */}
                <div className='form-group row'>
                    <label htmlFor="inputPassword" className='col-sm-2 col-form-label'>
                        Password
                    </label>
                    {/* getting the input */}
                    <div className='col-sm-10'>
                        {/* getting the user input */}
                        <input 
                        type='password' // type will show this ****** so won't show the actual password
                        // id is to locate which input field the label corresponds to
                        id="inputPassword"
                        className='form-control'
                        value={password}
                        onChange= {(event) => setPassword(event.target.value)}
                        />
                    </div>
                </div>

                {/* creating the Login button [ Login ] */}
                <div className='form-group row'>
                    <div className='col-sm-10'>
                        <button type='submit' className='btn btn-primary'>Login</button>
                    </div>
                </div>

                {/* if they don't have an account then will send them to register for one */}
                {/* again using bootstrap */}
                 <div className='row'>
                    <div className='col-sm-9 ml-sm-auto'>
                        {/* using Link since using React Router */}
                        <Link to="/register">Create an account</Link>
                    </div>
                </div>
            </form>
        </div>
    )
}