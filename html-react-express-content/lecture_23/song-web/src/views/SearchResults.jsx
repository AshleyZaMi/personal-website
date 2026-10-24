import { useEffect } from "react"
import { useState } from "react"
import { Link, useLocation, useNavigate } from "react-router-dom"

export default function SearchResults() {

	const [tracks, setTracks] = useState([
		{ trackId: 1, name: "For Those About To Rock (We Salute You)", genre: "Rock", mediaType: "MPEG audio file" },
		{ trackId: 2, name: "Balls to the Wall", genre: "Rock", mediaType: "Protected AAC audio file" }
	])

	const location = useLocation()
	const navigate = useNavigate()

	const fetchResults = async () => {

		const params = {
			trackName: location.state.trackInput,
			genreId: location.state.genreInput,
			mediaTypeId: location.state.mediaInput
		}

		const response = await fetch("http://localhost:3000/tracks/search?" + new URLSearchParams(params).toString())
		const results = await response.json()
		setTracks(results.tracks)
	}

	useEffect(() => {
		fetchResults()
	}, [])

	// create handler for delete button
	// have to pass track name to use it in here
	// has to take ann id as well
	const handleDelete = async(track, id) => {
		// thiere is a window which has ok or cancel. Will return true or false whether user clicked it or not
		const confirmed = window.confirm(`Are you sure you want to delete track ${track}`)
		
		// take a lok
		console.log(confirmed)

		// if confirmed is true want to delete stuff
		if(confirmed){
			// pass in id
			// fetch(endpoint, method type)
			const response = await fetch(`http://localhost:3000/tracks/${id}`, {
				method: "DELETE"
			}) //have to specify id

			// so lets pass in a status to Delete.jsx
			// extract status from response
			const status = response.status

			// navigate to the delete page to programatically navigate to another page
			// navigate(here the page, passing some data to the delete page )
			// passing this to the delete page
			// whether we succeed or not succeed will show that emssage
			navigate('/delete/',  {
				state:{
					track,
					status
				}
			})
		}

	}

	return (
		<>
			<ol className="breadcrumb p-3">
				<li className="breadcrumb-item"><Link to="/">Home</Link></li>
				<li className="breadcrumb-item"><Link to="/searchForm">Search</Link></li>
				<li className="breadcrumb-item active">Results</li>
			</ol>
			<div className="container">
				<div className="row">
					<h1 className="col-12 mt-4">Song Search Results</h1>
				</div>
			</div>
			<div className="container">
				<div className="row mb-4 mt-4">
					<div className="col-12">
						<Link to="/searchForm" role="button" className="btn btn-primary">Back to Form</Link>
					</div>
				</div>
				<div className="row">
					<div className="col-12">

						Showing {tracks.length} result(s).

					</div>
					<div className="col-12">
						<table className="table table-hover table-responsive mt-4">
							<thead>
								<tr>
									{/* create two more columns */}
									<th>Edit</th>
									<th>Delete</th>
									<th>Track</th>
									<th>Genre</th>
									<th>Media Type</th>
								</tr>
							</thead>
							<tbody>

								{
									// each track rendering a row
									tracks.map((row) => {
										return (

											<tr key={row.trackId}>
												{/* create a button */}
												{/* this one is to edit the form */}
												<td>
													<Link to={`/editForm/${row.trackId}`}
													className='btn btn-outline-warning'
												>
													Edit
													</Link>
												</td>

												{/* another table tadata for delete */}
												<td>
													<Link
													className = 'btn btn-outline-danger'
													// want to pass in the function 
													// each row know about the name
													// if user clicks ont hat then will have row.name of that specific row.
													// now update also so it takes it row.trackId
													onClick = {() => {handleDelete(row.name, row.trackId)}}
													>
														Delete
													</Link>
												</td>
												<td>
													<Link to={`/track/${row.trackId}`} >
														{row.name}
													</Link>
												</td>
												<td>{row.genre}</td>
												<td>{row.mediaType}</td>
											</tr>
										)
									})
								}

							</tbody>
						</table>
					</div>
				</div>
				<div className="row mt-4 mb-4">
					<div className="col-12">
						<Link to="/searchForm" role="button" className="btn btn-primary">Back to Form</Link>
					</div>
				</div>
			</div>

		</>
	)
}

