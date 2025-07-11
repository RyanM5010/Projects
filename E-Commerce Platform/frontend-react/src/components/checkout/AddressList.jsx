import React from 'react'
import { FaBuilding, FaCheckCircle, FaEdit, FaStreetView, FaTrash, FaTrashAlt } from 'react-icons/fa';
import { MdLocationCity, MdPinDrop, MdPublic } from 'react-icons/md';
import { useDispatch, useSelector } from 'react-redux'
import { selectUserCheckoutAddress } from '../../store/action';

const AddressList = ({addresses, setSelectedAddress, setOpenAddressModal, setOpenDeleteModal}) => {
    const dispatch = useDispatch();

    const {selectedUserCheckoutAddress} = useSelector((state)=> state.auth);

    const handleAddressSelection = (address)=>{
        dispatch(selectUserCheckoutAddress(address));
    };

    const onEditButtonHandler = (addresses) => {
        setSelectedAddress(addresses);
        setOpenAddressModal(true);
    };
    const onDeleteButtonHandler = (addresses) => {
        setSelectedAddress(addresses);
        setOpenDeleteModal(true);
    };

  return (
    <div className='space-y-4'>
        {addresses.map((address)=>(
            <div
                key={address.addressId} 
                onClick={()=>handleAddressSelection(address)}  
                className={`p-4 border rounded-2xl cursor-pointer relative ${
                    selectedUserCheckoutAddress?.addressId===address.addressId
                    ? "bg-blue-200" : "bg-white"
                }`}>
                <div className='flex items-start'>
                    <div className='space-y-1'>
                        <div className='flex items-center'>
                            <FaBuilding size={14} className='mr-2 text-gray-600'/>
                            <p className='font-semibold'>{address.buildingName}</p>
                            {selectedUserCheckoutAddress?.addressId===address.addressId && (
                                <FaCheckCircle className='text-blue-500 ml-2'/>
                            )}
                        </div>

                        <div className='flex items-center'>
                            <FaStreetView size={17} className='mr-2 text-gray-600'/>
                            <p>{address.street}</p>
                        </div>

                        <div className='flex items-center'>
                            <MdLocationCity size={17} className='mr-2 text-gray-600'/>
                            <p>{address.city}, {address.state}</p>
                        </div>

                        <div className='flex items-center'>
                            <MdPinDrop size={17} className='mr-2 text-gray-600'/>
                            <p>{address.zipCode}</p>
                        </div>

                        <div className='flex items-center'>
                            <MdPublic size={17} className='mr-2 text-gray-600'/>
                            <p>{address.country}</p>
                        </div>  
                    </div>
                </div> 

                <div className="flex gap-3 absolute top-4 right-2">
                    <button onClick={() => onEditButtonHandler(address)}>
                        <FaEdit size={18} className="text-teal-400" />
                    </button>
                    <button onClick={() => onDeleteButtonHandler(address)}>
                        <FaTrashAlt size={17} className="text-slate-400" />
                    </button>
                </div>                
            </div>
        ))}
        
    </div>
  )
}

export default AddressList