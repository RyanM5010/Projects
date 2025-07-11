import { Hourglass } from "react-loader-spinner";




const Loader = ({text}) => {
    return (
        <div className="flex justify-center items-center w-full h-[450px]">
            <div className="flex flex-col items-center gap-1">
            <Hourglass
                visible={true}
                height="80"
                width="80"
                ariaLabel="hourglass-loading"
                wrapperStyle={{}}
                wrapperClass=""
                colors={['#306cce', '#72a1ed']}
                />
                <p className="text-slate-800">
                    {text ? text : "Please wait.."}
                    </p>

            </div>
        </div>    
    )
}


export default Loader;