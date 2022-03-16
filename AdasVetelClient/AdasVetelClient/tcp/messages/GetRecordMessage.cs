namespace AdasVetelServer.messages
{
    class GetRecordMessage : MessageBase
    {
        public string Route { get; set; }
        public byte[] Record { get; set; }
        public int Index { get; set; }

        public GetRecordMessage(string route, byte[] record, int index, string sessionId) : base("RecordRequest", sessionId)
        {
            Route = route;
            Record = record;
            Index = index;
        }
    }
}
