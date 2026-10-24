import {useState} from 'react'
import {Link, useNavigate} from 'react-router-dom'
import '../styles/RegisterForm.css'

// will have a live one now so will replace to this path
const API_BASE_URL = import.meta.env.VITE_API_URL

export default function RegisterForm(){

    const navigate = useNavigate()

    // form states
    const [username, setUsername] = useState('')
    const [email, setEmail] = useState('')
    const [password, setPassword] = useState('')

    // validation errors
    const [errors, setErrors] = useState({})
    //backend error
    const [serverError, setServerError] = useState('')

    const handleSubmit = async(event) =>{
        event.preventDefault()

        // reseting errors
        setErrors({})
        setServerError('')

        const newErrors={}

        // front end validation
        if(!username.trim()){
            newErrors.username = 'Username is required'
        }

        if(!email.trim()){
            newErrors.email = 'Email is required'
        }

        if(!password.trim()){
            newErrors.password = "Password is required"
        }

        // if there are errors will stop submission
        if(Object.keys(newErrors).length > 0){
            setErrors(newErrors)
            return
        }
        try{
            // send request to backend
            const response = await fetch(API_BASE_URL+'/register', {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({
                    username,
                    email,
                    password
                }),
                credentials: "include"
            })

            const result = await response.json()
            console.log(result)
            // backend errors
            if(!response.ok){
                setServerError(result.message)
                return
            }

            //sucess
            navigate('/login')
        }catch(error){
            console.log(error)
        }
    }
    return(
    <div className='bootstrap-form-wrapper additional-reg-form-wrapper-styling'>
        <h2>Register</h2>
        {/* // using bootstrap for the form */}
            <form onSubmit={handleSubmit}>
                {/* handling the back end errors */}
                <div className='row mb-3'>
                    <div className='font-italic text-danger col-sm-9 ml-sm-auto'>
                        {serverError && serverError}
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
                        className={`form-control ${
                            errors.username ? 'is-invalid' : ''
                        }`} //for showing is errors or not dynamically
                        value={username}
                        onChange= {(event) => setUsername(event.target.value)}
                        />

                        {/* adding the backend API given errors dynamically, shows div if there is
                        errors*/}
                        {errors.username && (
                            <div className='text-danger'>
                                {errors.username}
                            </div>
                        )}
                    </div>
                </div>

                {/* form group row for email */}
                {/* form group row specifically for username*/}
                <div className='form-group row'>
                    {/* col-form-;abe; is so that they are vertically centered with their associated controls */}
                    <label htmlFor="inputEmail" className='col-sm-2 col-form-label'>
                        Email
                    </label>
                    {/* getting the input */}
                    <div className='col-sm-10'>
                        {/* getting the user input */}
                        <input 
                        type='email'
                        // id is to locate which input field the label corresponds to
                        id="inputEmail"
                        className={`form-control ${
                            errors.email ? 'is-invalid' : ''
                        }`} //for showing is errors or not dynamically
                        value={email}
                        onChange= {(event) => setEmail(event.target.value)}
                        />
                          {/* adding the backend API given errors dynamically*/}
                        {errors.email && (
                            <div className='text-danger'>
                                {errors.email}
                            </div>
                        )}
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
                        className={`form-control ${
                            errors.password ? 'is-invalid' : ''
                        }`} //for showing is errors or not dynamically
                        value={password}
                        onChange= {(event) => setPassword(event.target.value)}
                        />
                          {/* adding the backend API given errors dynamically*/}
                        {errors.password && (
                            <div className='text-danger'>
                                {errors.password}
                            </div>
                        )}
                    </div>
                </div>

                {/* creating the Register button [ Register ] */}
                <div className='form-group row'>
                    <div className='col-sm-10'>
                        <button type='submit' className='btn btn-primary'>Register</button>
                    </div>
                </div>

                {/* if they don't have an account then will send them to register for one */}
                {/* again using bootstrap */}
                <div className='row'>
                    <div className='col-sm-9 ml-sm-auto'>
                        {/* using Link since using React Router */}
                        <Link to="/login">Already have an account?</Link>
                    </div>
                </div>
            </form>
        </div>
    )
}
