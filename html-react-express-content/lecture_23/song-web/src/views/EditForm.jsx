import { useState, useEffect } from "react"
import { Link, useNavigate, useParams } from "react-router-dom"
import {fetchGenres} from '../fetch/index.js'

export default function EditForm() {
	// every time. state changes the component is goign to re rerender
	const [genres, setGenres] = useState([
		{ genreId: 1, name: 'Rock' },
		{ genreId: 2, name: 'Jazz' }
	])
	const [trackInput, setTrackInput] = useState('')
	const [genreInput, setGenreInput] = useState('')
	const [composerInput, setComposerInput] = useState('')
	const navigate = useNavigate()
	const params = useParams()

	// fetch track informaiton
	const fetchTrack = async() =>{
		// to get the id from params
		const response = await fetch(`http://localhost:3000/tracks/${params.id}`)
		const result = await response.json()
		setTrackInput(result.name)
		setComposerInput(result.composer)
		setGenreInput(result.genreId)
	}

	const getGenres = async () =>{
		setGenres(await fetchGenres())
	}

	useEffect(()=>{
		fetchTrack()
		getGenres()
	}, []) //array empty so it only runs in the beging, but if we want to run based on one state we could put it in the array

	const handleSubmit = async (event) => {
		event.preventDefault()

		// pass in the fields
		const fields = {
			name: trackInput,
			genreId: genreInput,
			composer: composerInput
		}

		// lets call on this patch 
		const response = await fetch(`http://localhost:3000/tracks/${params.id}`,{
				method: "PATCH",
				body: new URLSearchParams(fields)
			}
		)

		const result = await response.json()

		navigate('/editConfirm', {
			state: {
				status: response.status,
				message: result.message,
				trackName: trackInput
			}
		})
	}

	return (
		<>
			<ol className="breadcrumb p-3">
				<li className="breadcrumb-item"><Link to="/">Home</Link></li>
				<li className="breadcrumb-item"><Link to="/searchForm">Search</Link></li>
				<li className="breadcrumb-item"><Link to="/searchResults">Results</Link></li>
				<li className="breadcrumb-item active">Edit</li>
			</ol>
			<div className="container">
				<div className="row">
					<h1 className="col-12 mt-4 mb-4">Edit a Song</h1>
				</div>
			</div>
			<div className="container">

				<form onSubmit={handleSubmit}>

					<div className="form-group row mb-3">
						<label htmlFor="name-id" className="col-sm-3 col-form-label text-sm-right">
							Track Name: <span className="text-danger">*</span>
						</label>
						<div className="col-sm-9">
							<input value={trackInput} onChange={(e) => setTrackInput(e.target.value)} type="text" className="form-control" id="name-id" name="trackName" />
						</div>
					</div>

					<div className="form-group row mb-3">
						<label htmlFor="genre-id" className="col-sm-3 col-form-label text-sm-right">
							Genre: <span className="text-danger">*</span>
						</label>
						<div className="col-sm-9">
							<select value={genreInput} onChange={(e) => setGenreInput(e.target.value)} name="genreId" id="genre-id" className="form-control">
								<option value='' disabled>-- Select One --</option>

								{genres.map((row) => {
									return (
										<option key={row.genreId} value={row.genreId}>
											{row.name}
										</option>
									)
								})}Í

							</select>
						</div>
					</div>

					<div className="form-group row mb-3">
						<label htmlFor="composer-id" className="col-sm-3 col-form-label text-sm-right">
							Composer:
						</label>
						<div className="col-sm-9">
							<input value={composerInput ? composerInput : ''} onChange={(e) => setComposerInput(e.target.value)} type="text" name="composer" id="composer-id" className="form-control" />
						</div>
					</div>

					<div className="form-group row mb-3">
						<div className="ml-auto col-sm-9">
							<span className="text-danger font-italic">* Required</span>
						</div>
					</div>

					<div className="form-group row">
						<div className="col-sm-3"></div>
						<div className="col-sm-9 mt-2">
							<button type="submit" className="btn btn-primary">Submit</button>
							<button type="reset" className="btn btn-light">Reset</button>
						</div>
					</div>
				</form>
			</div>
		</>
	)
}