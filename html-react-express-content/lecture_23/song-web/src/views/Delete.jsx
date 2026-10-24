import { Link , useLocation} from "react-router-dom"
// want to use lovation,
//in delete compoenent can have access to the location oject
export default function () {
	// if pass information from navigate, so we want to use that recieved data__ with useLocation
	const location = useLocation() //so in location have access to state track and status
	return (
		<>
			<ol className="breadcrumb p-3">
				<li className="breadcrumb-item"><Link to="/">Home</Link></li>
				<li className="breadcrumb-item"><Link to="/searchForm">Search</Link></li>
				<li className="breadcrumb-item"><Link to="/searchResults">Results</Link></li>
				<li className="breadcrumb-item active">Delete</li>
			</ol>
			<div className="container">
				<div className="row">
					<h1 className="col-12 mt-4">Delete a Song</h1>
				</div>
			</div>
			<div className="container">
				<div className="row mt-4">
					<div className="col-12">
					{/* check on status */}
						{
							location.state.status == 200 ?
								// <div className='text-success'>
								// 	<span>Track Name</span> has been successfully deleted.
								// </div>
								<div className='text-success'>
									<span>{location.state.track}</span> has been successfully deleted.
								</div>
								:
								// <div className='text-danger'>
								// 	<span>Track Name</span> has not been successfully deleted.
								// </div>
								<div className='text-danger'>
									<span>{location.state.track}</span> has not been successfully deleted.
								</div>
						}

							{/* <div className='text-success'>
								<span>Track Name</span> has been successfully deleted.
							</div> */}

					</div>
				</div>
				<div className="row mt-4 mb-4">
					<div className="col-12">
						<Link to="/searchForm" role="button" className="btn btn-primary">Back to Search Form</Link>
					</div>
				</div>
			</div>
		</>
	)
}
